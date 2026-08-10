"""
Cleans dataset_v16v2_200726.jsonl for LLM fine-tuning.

Fixes applied (see analysis notes for how each was found):

1. Schema normalization  - 106 records use an abbreviated key schema
   (tt/r/c/cs, v/cf, rt/cr/nr/d/sn, ct/cc, p/cid/cm/u/s/sm) instead of the
   dominant full-name schema (table_type/rows/columns/cells, value/confidence,
   row_type/contributing_rows/note_ref_value/direction/is_signed_negative,
   column_type/contributing_columns, reporting_period/concept_id/
   concept_meaning/unit/scale/scale_multiplier). Both are expanded to the
   same canonical full-name schema so the model only ever sees one output
   format.

2. Structural repair - a handful of records have stray top-level numeric
   keys (e.g. `"2": {...}`) that are cell data which escaped the `cells`
   dict during label generation. These are merged back into `cells`
   (existing `cells` entries win on conflict) instead of being silently
   dropped or left as invalid top-level junk keys.

3. Degenerate-record removal - records where `table_type.value` is null
   after normalization (a genuine generation failure, not a real "None"
   class) are dropped.

4. Deduplication - 176 groups (355 records) share an identical prompt but
   carry different completions (two independent label passes over the same
   source table were both included in the export). Same input -> different
   "correct" output is pure label noise, so only the single best-scoring
   completion per unique prompt is kept (scored by mean confidence across
   table_type/rows/columns/cells; ties keep the first occurrence).

5. Light text normalization - `table_type.value` strings get whitespace
   collapsed/stripped only (no case changes, no semantic merging). The
   ~2,235-way fragmentation of table_type strings (e.g. "balance_sheet" vs
   "statement of financial position") is a semantic clustering problem, not
   a mechanical bug, so it is intentionally NOT auto-merged here - do that
   as a separate, reviewed step if you want a controlled taxonomy.

Anything that still doesn't parse/validate after all of the above is
dropped and logged rather than silently kept or crashing the run.
"""

import json
import os
import re
import statistics
from collections import defaultdict

# ==========================================
# CONFIGURATION
# ==========================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(ROOT_DIR, "dataset_v16v2_200726.jsonl")
OUTPUT_PATH = os.path.join(ROOT_DIR, "dataset_v16v2_200726.clean.jsonl")
REPORT_PATH = os.path.join(ROOT_DIR, "dataset_clean_report.txt")

INPUT_MARKER = "### Input:\n"
RESPONSE_MARKER = "\n\n### Response:"

# tt/r/c/cs abbreviation -> full name maps, reverse-engineered from the
# 106 abbreviated records (verified consistent across all of them).
TT_MAP = {"v": "value", "cf": "confidence"}
ROW_MAP = {
    "rt": "row_type",
    "cr": "contributing_rows",
    "nr": "note_ref_value",
    "d": "direction",
    "sn": "is_signed_negative",
    "cf": "confidence",
}
COL_MAP = {
    "ct": "column_type",
    "cc": "contributing_columns",
    "d": "direction",
    "cf": "confidence",
}
CELL_MAP = {
    "p": "reporting_period",
    "cid": "concept_id",
    "cm": "concept_meaning",
    "u": "unit",
    "s": "scale",
    "sm": "scale_multiplier",
    "cf": "confidence",
}

CANONICAL_KEYS = {"table_type", "rows", "columns", "cells"}
ABBREV_KEYS = {"tt", "r", "c", "cs"}


def _expand(d, key_map):
    return {key_map.get(k, k): v for k, v in d.items()}


def expand_abbreviated_completion(comp):
    """Expand the tt/r/c/cs abbreviated schema into the canonical schema."""
    out = {}
    out["table_type"] = _expand(comp["tt"], TT_MAP)
    out["rows"] = {ridx: _expand(rv, ROW_MAP) for ridx, rv in comp["r"].items()}
    out["columns"] = {cidx: _expand(cv, COL_MAP) for cidx, cv in comp["c"].items()}
    out["cells"] = {
        ridx: {cidx: _expand(cv, CELL_MAP) for cidx, cv in rowdict.items()}
        for ridx, rowdict in comp["cs"].items()
    }
    return out


def repair_stray_keys(comp, stats):
    """Merge stray top-level numeric keys (leaked row-cell data) into `cells`."""
    stray_keys = [k for k in comp.keys() if k.isdigit()]
    if not stray_keys:
        return comp
    cells = comp.setdefault("cells", {})
    for k in stray_keys:
        stray_value = comp.pop(k)
        if k not in cells:
            cells[k] = stray_value
            stats["stray_keys_recovered"] += 1
        elif cells[k] == stray_value:
            stats["stray_keys_dropped_duplicate"] += 1
        else:
            # Existing `cells` entry is treated as authoritative; the stray
            # sibling is discarded but counted so it's visible in the report.
            stats["stray_keys_dropped_conflict"] += 1
    return comp


def normalize_completion(comp, stats):
    keys = set(comp.keys())

    if keys == ABBREV_KEYS:
        stats["abbreviated_schema_expanded"] += 1
        return expand_abbreviated_completion(comp)

    if CANONICAL_KEYS.issubset(keys):
        if keys - CANONICAL_KEYS:
            comp = repair_stray_keys(comp, stats)
        return comp

    stats["unknown_schema_dropped"] += 1
    return None


def clean_table_type_text(value):
    if not isinstance(value, str):
        return value
    return re.sub(r"\s+", " ", value).strip()


def collect_confidences(comp):
    confidences = []

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "confidence" and isinstance(v, (int, float)):
                    confidences.append(v)
                else:
                    walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(comp)
    return confidences


def extract_input_json(prompt):
    start = prompt.index(INPUT_MARKER) + len(INPUT_MARKER)
    end = prompt.index(RESPONSE_MARKER)
    return json.loads(prompt[start:end])


def clean_dataset():
    stats = defaultdict(int)
    kept = {}  # prompt_hash -> (score, index, prompt, comp)
    conflict_log = []

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            stats["total_input_records"] += 1

            try:
                record = json.loads(line)
                prompt = record["prompt"]
                comp = json.loads(record["completion"])
            except Exception:
                stats["dropped_parse_error"] += 1
                continue

            try:
                inp = extract_input_json(prompt)
            except Exception:
                stats["dropped_unparseable_input"] += 1
                continue

            comp = normalize_completion(comp, stats)
            if comp is None:
                continue

            table_type = comp.get("table_type", {})
            table_type["value"] = clean_table_type_text(table_type.get("value"))
            if table_type.get("value") is None:
                stats["dropped_null_table_type"] += 1
                continue

            classify_rows = inp.get("classify_rows", [])
            classify_cols = inp.get("classify_columns", [])
            if len(comp.get("rows", {})) != len(classify_rows) or len(
                comp.get("columns", {})
            ) != len(classify_cols):
                stats["dropped_row_col_count_mismatch"] += 1
                continue

            confidences = collect_confidences(comp)
            score = statistics.mean(confidences) if confidences else 0.0

            prompt_hash = hash(prompt)
            new_completion = json.dumps(comp, ensure_ascii=False)

            existing = kept.get(prompt_hash)
            if existing is None:
                kept[prompt_hash] = (score, lineno, prompt, new_completion, table_type["value"])
            else:
                stats["duplicate_prompt_conflicts"] += 1
                if existing[3] == new_completion:
                    stats["duplicate_prompt_exact_dupe"] += 1
                else:
                    conflict_log.append(
                        f"line {lineno} vs line {existing[1]}: "
                        f"kept table_type={existing[4]!r} (score={existing[0]:.3f}) "
                        f"over table_type={table_type['value']!r} (score={score:.3f})"
                    )
                if score > existing[0]:
                    kept[prompt_hash] = (
                        score,
                        lineno,
                        prompt,
                        new_completion,
                        table_type["value"],
                    )

    written = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for score, lineno, prompt, completion, _ in sorted(kept.values(), key=lambda x: x[1]):
            out.write(json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False) + "\n")
            written += 1

    stats["final_records_written"] = written

    report_lines = ["=== Dataset cleaning report ===", ""]
    for key in [
        "total_input_records",
        "dropped_parse_error",
        "dropped_unparseable_input",
        "abbreviated_schema_expanded",
        "stray_keys_recovered",
        "stray_keys_dropped_duplicate",
        "stray_keys_dropped_conflict",
        "unknown_schema_dropped",
        "dropped_null_table_type",
        "dropped_row_col_count_mismatch",
        "duplicate_prompt_conflicts",
        "duplicate_prompt_exact_dupe",
        "final_records_written",
    ]:
        report_lines.append(f"{key}: {stats.get(key, 0)}")

    report_lines.append("")
    report_lines.append(f"=== Duplicate-prompt conflicts resolved ({len(conflict_log)}) ===")
    report_lines.extend(conflict_log)

    report = "\n".join(report_lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as rf:
        rf.write(report)

    print(report)
    print(f"\nCleaned dataset written to '{OUTPUT_PATH}'")
    print(f"Full report written to '{REPORT_PATH}'")


if __name__ == "__main__":
    clean_dataset()
