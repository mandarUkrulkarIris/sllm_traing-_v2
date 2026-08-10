import json
import collections
import statistics

DATASET_PATH = r"D:\Dev\sllm_training_v2\dataset_v16v2_200726.clean.jsonl"
OUT_JSON = r"D:\Dev\sllm_training_v2\report_assets\label_stats.json"

table_type_counts = collections.Counter()
unit_counts = collections.Counter()
scale_counts = collections.Counter()
scale_multiplier_counts = collections.Counter()

# concept_id -> Counter of concept_meaning strings used for that concept_id
concept_id_meanings = collections.defaultdict(collections.Counter)
# case/whitespace-insensitive concept_id grouping -> set of raw variants seen
concept_id_casefold_groups = collections.defaultdict(collections.Counter)
# case-insensitive table_type grouping -> set of raw variants seen (near-duplicate detection)
table_type_casefold_groups = collections.defaultdict(collections.Counter)

n_records = 0
n_cells = 0
n_parse_errors = 0

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        n_records += 1
        try:
            rec = json.loads(line)
            comp = json.loads(rec["completion"])
        except Exception:
            n_parse_errors += 1
            continue

        tt = comp.get("table_type", {})
        tt_val = tt.get("value") if isinstance(tt, dict) else tt
        if tt_val:
            table_type_counts[tt_val] += 1
            table_type_casefold_groups[tt_val.strip().lower()][tt_val] += 1

        cells = comp.get("cells", {})
        for row_idx, row_cells in cells.items():
            if not isinstance(row_cells, dict):
                continue
            for col_idx, cell in row_cells.items():
                if not isinstance(cell, dict):
                    continue
                n_cells += 1
                cid = cell.get("concept_id")
                meaning = cell.get("concept_meaning")
                unit = cell.get("unit")
                scale = cell.get("scale")
                mult = cell.get("scale_multiplier")

                if unit is not None:
                    unit_counts[unit] += 1
                if scale is not None:
                    scale_counts[scale] += 1
                if mult is not None:
                    scale_multiplier_counts[str(mult)] += 1

                if cid:
                    if meaning:
                        concept_id_meanings[cid][meaning] += 1
                    concept_id_casefold_groups[cid.strip().lower()][cid] += 1

# ---- table_type imbalance ----
n_table_type_total = sum(table_type_counts.values())
n_unique_table_types = len(table_type_counts)
sorted_tt = table_type_counts.most_common()
top20_tt = sorted_tt[:20]
singleton_tt = [v for v, c in sorted_tt if c == 1]
doubleton_tt = [v for v, c in sorted_tt if c == 2]
top10_share = sum(c for _, c in sorted_tt[:10]) / n_table_type_total
top20_share = sum(c for _, c in sorted_tt[:20]) / n_table_type_total

# near-duplicate table_type values (differ only by case/whitespace)
tt_dupe_groups = {k: dict(v) for k, v in table_type_casefold_groups.items() if len(v) > 1}

# ---- concept_id / concept_meaning consistency ----
n_unique_concept_ids = len(concept_id_meanings)
variant_counts = {cid: len(meanings) for cid, meanings in concept_id_meanings.items()}
multi_variant_concepts = {cid: n for cid, n in variant_counts.items() if n > 1}
n_multi_variant = len(multi_variant_concepts)

worst_offenders = sorted(multi_variant_concepts.items(), key=lambda kv: kv[1], reverse=True)[:25]
worst_offender_detail = []
for cid, n in worst_offenders:
    worst_offender_detail.append({
        "concept_id": cid,
        "n_distinct_meanings": n,
        "total_occurrences": sum(concept_id_meanings[cid].values()),
        "top_meanings": concept_id_meanings[cid].most_common(5),
    })

# near-duplicate concept_id values (differ only by case/whitespace)
cid_dupe_groups = {k: dict(v) for k, v in concept_id_casefold_groups.items() if len(v) > 1}

variant_count_distribution = collections.Counter(variant_counts.values())

# ---- unit / scale / scale_multiplier consistency ----
unit_sorted = unit_counts.most_common()
scale_sorted = scale_counts.most_common()
mult_sorted = scale_multiplier_counts.most_common()

stats = {
    "n_records": n_records,
    "n_parse_errors": n_parse_errors,
    "n_cells_total": n_cells,
    "table_type": {
        "n_unique_values": n_unique_table_types,
        "n_total_labeled": n_table_type_total,
        "top20": top20_tt,
        "n_singleton_values": len(singleton_tt),
        "n_doubleton_values": len(doubleton_tt),
        "pct_values_appearing_once": len(singleton_tt) / n_unique_table_types if n_unique_table_types else None,
        "top10_share_of_examples": top10_share,
        "top20_share_of_examples": top20_share,
        "n_near_duplicate_groups_case_insensitive": len(tt_dupe_groups),
        "near_duplicate_groups_sample": dict(list(tt_dupe_groups.items())[:15]),
    },
    "concept_id_meaning_consistency": {
        "n_unique_concept_ids": n_unique_concept_ids,
        "n_concept_ids_with_multiple_meaning_variants": n_multi_variant,
        "pct_concept_ids_with_multiple_meaning_variants": n_multi_variant / n_unique_concept_ids if n_unique_concept_ids else None,
        "variant_count_distribution": dict(sorted(variant_count_distribution.items())),
        "worst_offenders": worst_offender_detail,
        "n_near_duplicate_concept_id_groups_case_insensitive": len(cid_dupe_groups),
        "near_duplicate_concept_id_groups_sample": dict(list(cid_dupe_groups.items())[:15]),
    },
    "unit": {
        "n_unique_values": len(unit_counts),
        "top30": unit_sorted[:30],
        "n_singleton_values": sum(1 for _, c in unit_sorted if c == 1),
    },
    "scale": {
        "n_unique_values": len(scale_counts),
        "all_values": scale_sorted,
    },
    "scale_multiplier": {
        "n_unique_values": len(scale_multiplier_counts),
        "all_values": mult_sorted,
    },
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)

print("n_records", n_records, "n_parse_errors", n_parse_errors, "n_cells_total", n_cells)
print()
print("=== table_type ===")
print("n_unique_values:", n_unique_table_types, "/ n_total_labeled:", n_table_type_total)
print("top10 share of examples:", f"{top10_share*100:.1f}%")
print("top20 share of examples:", f"{top20_share*100:.1f}%")
print("values appearing exactly once:", len(singleton_tt), f"({len(singleton_tt)/n_unique_table_types*100:.1f}% of unique values)")
print("values appearing exactly twice:", len(doubleton_tt))
print("near-duplicate (case/whitespace) groups:", len(tt_dupe_groups))
print("top 20 table_type values:")
for v, c in top20_tt:
    print(f"  {c:5d}  {v}")
print()
print("=== concept_id -> concept_meaning consistency ===")
print("n_unique_concept_ids:", n_unique_concept_ids)
print("concept_ids with >1 distinct meaning wording:", n_multi_variant,
      f"({n_multi_variant/n_unique_concept_ids*100:.1f}%)")
print("variant-count distribution (n_variants: n_concept_ids):", dict(sorted(variant_count_distribution.items())))
print("near-duplicate (case/whitespace) concept_id groups:", len(cid_dupe_groups))
print("worst offenders (most inconsistent wording):")
for w in worst_offender_detail[:10]:
    print(f"  {w['concept_id']}  ({w['n_distinct_meanings']} variants, {w['total_occurrences']} occurrences)")
    for m, c in w["top_meanings"]:
        print(f"      [{c:3d}x] {m}")
print()
print("=== unit ===")
print("n_unique_values:", len(unit_counts))
for v, c in unit_sorted[:30]:
    print(f"  {c:6d}  {v!r}")
print()
print("=== scale ===")
for v, c in scale_sorted:
    print(f"  {c:6d}  {v!r}")
print()
print("=== scale_multiplier ===")
for v, c in mult_sorted:
    print(f"  {c:6d}  {v!r}")
print()
print("Wrote", OUT_JSON)
