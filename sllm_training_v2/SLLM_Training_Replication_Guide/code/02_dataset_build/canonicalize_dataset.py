import json
import re
import collections

INPUT_PATH = r"D:\Dev\sllm_training_v2\dataset_v16v2_200726.clean.jsonl"
OUTPUT_PATH = r"D:\Dev\sllm_training_v2\dataset_v16v2_200726.clean.canonical.jsonl"
REPORT_PATH = r"D:\Dev\sllm_training_v2\report_assets\canonicalization_report.json"

# A concept_id's meaning is only auto-canonicalized (collapsed to its single most
# common wording) if one wording already dominates its occurrences. Below this
# threshold, the variation is treated as a signal the concept_id itself is too
# generic (e.g. "Total") rather than as paraphrase noise, and is left untouched
# so genuinely different meanings aren't silently overwritten with the wrong text.
CONCEPT_MEANING_DOMINANCE_THRESHOLD = 0.6

# Known unit synonyms found by analyze_dataset_labels.py that aren't caught by
# the generic case/whitespace/underscore fold below (different words entirely).
UNIT_ALIAS_MAP = {
    "percentage": "percent",
    "EUR_thousand": "thousand EUR",
    "thousand euro": "thousand EUR",
}

SCALE_WORDS = {"thousand", "million", "billion", "hundred", "cents", "cent"}


def fold_key(s):
    return re.sub(r"\s+", " ", s.strip().lower().replace("_", " "))


def normalize_multiplier(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return int(f) if f == int(f) else f


def looks_like_mixed_scale_unit(unit_value):
    key = fold_key(unit_value)
    tokens = key.split(" ")
    return len(tokens) > 1 and any(t in SCALE_WORDS for t in tokens)


# ==========================================
# PASS 1 — build canonicalization maps from the whole dataset
# ==========================================
table_type_fold_groups = collections.defaultdict(collections.Counter)
concept_id_fold_groups = collections.defaultdict(collections.Counter)
concept_id_meanings = collections.defaultdict(collections.Counter)
unit_fold_groups = collections.defaultdict(collections.Counter)

n_records = 0
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        n_records += 1
        rec = json.loads(line)
        comp = json.loads(rec["completion"])

        tt = comp.get("table_type", {})
        tt_val = tt.get("value") if isinstance(tt, dict) else tt
        if tt_val:
            table_type_fold_groups[fold_key(tt_val)][tt_val] += 1

        for row_cells in comp.get("cells", {}).values():
            if not isinstance(row_cells, dict):
                continue
            for cell in row_cells.values():
                if not isinstance(cell, dict):
                    continue
                cid = cell.get("concept_id")
                meaning = cell.get("concept_meaning")
                unit = cell.get("unit")

                if cid:
                    concept_id_fold_groups[fold_key(cid)][cid] += 1
                    if meaning:
                        concept_id_meanings[cid][meaning] += 1
                if unit:
                    unit_fold_groups[fold_key(unit)][unit] += 1

table_type_canon_map = {k: v.most_common(1)[0][0] for k, v in table_type_fold_groups.items()}
concept_id_canon_map = {k: v.most_common(1)[0][0] for k, v in concept_id_fold_groups.items()}

unit_canon_map = {k: v.most_common(1)[0][0] for k, v in unit_fold_groups.items()}
for raw, alias_target in UNIT_ALIAS_MAP.items():
    unit_canon_map[fold_key(raw)] = alias_target

concept_meaning_canon_map = {}
ambiguous_concept_ids = []
for cid, meanings in concept_id_meanings.items():
    if len(meanings) <= 1:
        continue
    total = sum(meanings.values())
    top_meaning, top_count = meanings.most_common(1)[0]
    dominance = top_count / total
    if dominance >= CONCEPT_MEANING_DOMINANCE_THRESHOLD:
        concept_meaning_canon_map[cid] = top_meaning
    else:
        ambiguous_concept_ids.append({
            "concept_id": cid,
            "n_variants": len(meanings),
            "total_occurrences": total,
            "top_variant_share": dominance,
            "top_meanings": meanings.most_common(5),
        })

ambiguous_concept_ids.sort(key=lambda d: d["total_occurrences"], reverse=True)
ambiguous_concept_id_set = {d["concept_id"] for d in ambiguous_concept_ids}

mixed_scale_units = collections.Counter()
for variants in unit_fold_groups.values():
    for raw, count in variants.items():
        if looks_like_mixed_scale_unit(raw):
            mixed_scale_units[raw] += count

print("Pass 1 done.")
print(f"  table_type: {len(table_type_fold_groups)} fold-groups from {sum(len(v) for v in table_type_fold_groups.values())} raw values")
print(f"  concept_id: {len(concept_id_fold_groups)} fold-groups")
print(f"  concept_meaning: {len(concept_meaning_canon_map)} concept_ids canonicalized, {len(ambiguous_concept_ids)} left ambiguous (not touched)")
print(f"  unit: {len(unit_fold_groups)} fold-groups, {len(mixed_scale_units)} distinct values flagged as mixed-scale (not auto-transformed)")

# ==========================================
# PASS 2 — rewrite the dataset applying the canonicalizations
# ==========================================
counts = {
    "table_type_remapped": 0,
    "concept_id_remapped": 0,
    "concept_meaning_remapped": 0,
    "unit_remapped": 0,
    "scale_multiplier_remapped": 0,
}

table_type_values_out = collections.Counter()
concept_id_values_out = collections.Counter()
unit_values_out = collections.Counter()
mult_values_out = collections.Counter()

with open(INPUT_PATH, "r", encoding="utf-8") as fin, open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        comp = json.loads(rec["completion"])

        tt = comp.get("table_type")
        if isinstance(tt, dict) and tt.get("value"):
            old = tt["value"]
            new = table_type_canon_map.get(fold_key(old), old)
            if new != old:
                counts["table_type_remapped"] += 1
            tt["value"] = new
            table_type_values_out[new] += 1

        for row_cells in comp.get("cells", {}).values():
            if not isinstance(row_cells, dict):
                continue
            for cell in row_cells.values():
                if not isinstance(cell, dict):
                    continue

                cid = cell.get("concept_id")
                if cid:
                    new_cid = concept_id_canon_map.get(fold_key(cid), cid)
                    if new_cid != cid:
                        counts["concept_id_remapped"] += 1
                    cell["concept_id"] = new_cid
                    concept_id_values_out[new_cid] += 1

                    meaning = cell.get("concept_meaning")
                    # Note: canonicalization keys off the ORIGINAL concept_id (pre-remap)
                    # since that's what the meaning map was built against.
                    if meaning and cid in concept_meaning_canon_map:
                        new_meaning = concept_meaning_canon_map[cid]
                        if new_meaning != meaning:
                            counts["concept_meaning_remapped"] += 1
                        cell["concept_meaning"] = new_meaning

                unit = cell.get("unit")
                if unit:
                    new_unit = unit_canon_map.get(fold_key(unit), unit)
                    if new_unit != unit:
                        counts["unit_remapped"] += 1
                    cell["unit"] = new_unit
                    unit_values_out[new_unit] += 1

                if "scale_multiplier" in cell and cell["scale_multiplier"] is not None:
                    old_mult = cell["scale_multiplier"]
                    new_mult = normalize_multiplier(old_mult)
                    if new_mult != old_mult or str(new_mult) != str(old_mult):
                        counts["scale_multiplier_remapped"] += 1
                    cell["scale_multiplier"] = new_mult
                    mult_values_out[str(new_mult)] += 1

        rec_out = {
            "prompt": rec["prompt"],
            "completion": json.dumps(comp, ensure_ascii=False),
        }
        fout.write(json.dumps(rec_out, ensure_ascii=False) + "\n")

report = {
    "n_records": n_records,
    "concept_meaning_dominance_threshold": CONCEPT_MEANING_DOMINANCE_THRESHOLD,
    "counts_of_cells_or_examples_changed": counts,
    "table_type": {
        "n_unique_before": sum(len(v) for v in table_type_fold_groups.values()) and len(
            {v for grp in table_type_fold_groups.values() for v in grp}
        ),
        "n_unique_after": len(table_type_values_out),
    },
    "concept_id": {
        "n_unique_before": len({v for grp in concept_id_fold_groups.values() for v in grp}),
        "n_unique_after": len(concept_id_values_out),
    },
    "unit": {
        "n_unique_before": len({v for grp in unit_fold_groups.values() for v in grp}),
        "n_unique_after": len(unit_values_out),
        "alias_map_applied": UNIT_ALIAS_MAP,
        "flagged_mixed_scale_units_not_auto_fixed": mixed_scale_units.most_common(),
    },
    "scale_multiplier": {
        "n_unique_before": None,  # not tracked pre-normalization in this pass; see analyze_dataset_labels.py output
        "n_unique_after": len(mult_values_out),
        "values_after": mult_values_out.most_common(),
    },
    "concept_meaning": {
        "n_concept_ids_canonicalized": len(concept_meaning_canon_map),
        "n_concept_ids_left_ambiguous": len(ambiguous_concept_ids),
        "top_ambiguous_concept_ids": ambiguous_concept_ids[:30],
    },
}

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print()
print("Pass 2 done. Wrote", OUTPUT_PATH)
print()
print("=== changes applied ===")
for k, v in counts.items():
    print(f"  {k}: {v}")
print()
print("=== unique-value counts, before -> after ===")
print(f"  table_type: {report['table_type']['n_unique_before']} -> {report['table_type']['n_unique_after']}")
print(f"  concept_id: {report['concept_id']['n_unique_before']} -> {report['concept_id']['n_unique_after']}")
print(f"  unit: {report['unit']['n_unique_before']} -> {report['unit']['n_unique_after']}")
print()
print(f"concept_meaning: canonicalized {len(concept_meaning_canon_map)} concept_ids; "
      f"left {len(ambiguous_concept_ids)} ambiguous concept_ids untouched (see report)")
print("top 10 ambiguous concept_ids (not canonicalized, needs manual review):")
for d in ambiguous_concept_ids[:10]:
    print(f"  {d['concept_id']}  ({d['n_variants']} variants, {d['total_occurrences']} occurrences, "
          f"top variant share {d['top_variant_share']*100:.0f}%)")
print()
print("unit values flagged as mixing scale into the unit string (not auto-fixed, needs manual review):")
for v, c in mixed_scale_units.most_common(15):
    print(f"  {c:6d}  {v!r}")
print()
print("Wrote report to", REPORT_PATH)
