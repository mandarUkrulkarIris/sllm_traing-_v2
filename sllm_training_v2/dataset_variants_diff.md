# Dataset Variants — `dataset_v16v2_200726` Family

Three files, three stages of the same pipeline. Each stage only *removes noise or normalizes values* — none of them touch the underlying labeling (still the original Azure/GPT-4.1 teacher labels).

| File | Stage | Records | Size | Produced by |
|---|---|---|---|---|
| `dataset_v16v2_200726.jsonl` | Raw compiled | 6,289 | 107,094,348 bytes (~102 MB) | `dataprep_140726.py` — reconciles exactly with the 6,289 source `_input`/`_response.json` pairs |
| `dataset_v16v2_200726.clean.jsonl` | Cleaned | 6,109 | 105,425,544 bytes (~100.5 MB) | `dataprep/dataset_clean.py` |
| `dataset_v16v2_200726.clean.canonical.jsonl` | Cleaned + canonicalized | 6,109 | 105,292,065 bytes (~100.4 MB) | `dataprep/canonicalize_dataset.py` |

**Which was actually used for training:** the production LoRA adapter documented in `Dataset_and_Training_Report.docx` (`teamspace_uploads_Qwen3.5-4B_v16v2_clean_dataset_220726_3epochs`) trained on **`.clean.jsonl`**. A second training script, `VM_training_scripts/main_v2.py` — the one carrying the final `r=32/alpha=64/target_modules="all-linear"` LoRA recipe — points at **`.clean.canonical.jsonl`** instead.

---

## 1. Raw → Clean: `dataset_clean.py`

Record count drops **6,289 → 6,109** (180 records removed, 2.9%). Nothing here changes label *values* — this stage only fixes structural/formatting problems and removes genuine noise:

| Fix | What it means | Count |
|---|---|---|
| Schema normalization | 106 records used an abbreviated key schema (`tt/r/c/cs`, `v/cf`, `rt/cr/nr/d/sn`, `ct/cc`, `p/cid/cm/u/s/sm`) instead of the dominant full-name schema — both expanded to one canonical schema so the model only ever sees one output format. | 106 records |
| Structural repair | Stray top-level numeric keys (cell data that escaped the `cells` dict during label generation) merged back into `cells`. | 11 recovered, 7 dropped as exact duplicates, 0 conflicts |
| Degenerate-record removal | Records where `table_type.value` is null after normalization (a genuine generation failure) are dropped. | 1 record dropped |
| **Deduplication** | 179 groups shared an identical prompt but carried two different completions (two independent label passes over the same source table both made it into the export). Same input → different "correct" output is label noise, so only the higher mean-confidence completion per prompt is kept. | 179 conflicts resolved, 0 exact dupes |
| Light text normalization | `table_type.value` strings get whitespace collapsed/stripped only — no case changes, no semantic merging. | applied dataset-wide |
| **Net result** | 6,289 → 6,109 records (**1** null-`table_type` drop **+ 179** duplicate-prompt conflicts resolved) | **−180 records** |

**Deliberately NOT done at this stage:** `table_type.value` still has ~2,232 distinct strings across near-duplicate wordings (e.g. `balance_sheet` vs. `balance sheet`) — the cleaning script's own docstring calls this "a semantic clustering problem, not a mechanical bug," and defers it to a separate, reviewed step. That step is §2.

---

## 2. Clean → Clean+Canonical: `canonicalize_dataset.py`

Record count is **unchanged (6,109 → 6,109)** — this stage relabels *values* for consistency, it doesn't add or drop records. Method: fold every value to a case/whitespace/underscore-insensitive key, then replace every variant with whichever exact wording is most common within that fold-group (plus a small manual alias map for units).

| Field | Unique values before | Unique values after | Cells/examples remapped |
|---|---|---|---|
| `table_type` | 2,232 | 2,087 | 420 |
| `concept_id` | 67,473 | 67,134 | 820 |
| `unit` | 210 | 188 | 1,175 |
| `concept_meaning` | — (keyed off `concept_id`, not counted standalone) | — | 2,742 |
| `scale_multiplier` | not tracked pre-normalization | 10 distinct final values | 64,730 (numeric-format normalization, e.g. `"1000.0"` → `1000`) |

**Guardrails built into this pass (why it doesn't just collapse everything):**
- `concept_meaning` is only auto-canonicalized for a `concept_id` if one wording already accounts for **≥60%** of its occurrences ("dominance threshold"). Below that, the variation is treated as a sign the `concept_id` itself is too generic (e.g. `"Total"`) rather than paraphrase noise, and is left untouched. Result: **1,676** concept_ids canonicalized, **17,211** left deliberately ambiguous.
- `unit` values that mix a scale word into the unit string itself (e.g. `"million HUF"`, `"EUR_thousand"`, `"billion_cubic_meter"`) are flagged in the report but **not auto-fixed** — collapsing those would require deciding whether the scale belongs in `unit` or `scale_multiplier`, which is a judgment call left for manual review. ~40 such variants are flagged, the largest being `million HUF` (322 occurrences).
- A small explicit alias map handles cases the generic fold can't catch because the words differ entirely: `percentage`→`percent`, `EUR_thousand`→`thousand EUR`, `thousand euro`→`thousand EUR`.

**Net effect:** the canonical file is a drop-in replacement for `.clean.jsonl` (same record count, same schema) with less spurious label fragmentation — most useful for the `table_type` field, where the open-vocabulary taxonomy is the hardest part of the task per the evaluation report (§3 of `Evaluation_and_Metrics_Report.docx`).

---

## 3. Summary

| | Raw (`.jsonl`) | Clean (`.clean.jsonl`) | Clean+Canonical (`.clean.canonical.jsonl`) |
|---|---|---|---|
| Records | 6,289 | 6,109 | 6,109 |
| Schema | Mixed (full-name + 106 abbreviated) | Single canonical schema | Same as Clean |
| Duplicate/conflicting prompts | Present (179 conflicts) | Resolved | Same as Clean |
| `table_type` distinct values | 2,232 (untouched) | 2,232 (untouched) | 2,087 |
| `concept_id` distinct values | 67,473 (untouched) | 67,473 (untouched) | 67,134 |
| `unit` distinct values | 210 (untouched) | 210 (untouched) | 188 |
| Used for production training run | No | **Yes** (`teamspace_uploads_Qwen3.5-4B_v16v2_clean_dataset_220726_3epochs`) | **Yes** (`main_v2.py`, the `all-linear` LoRA recipe) |
