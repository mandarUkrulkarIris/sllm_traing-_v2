# Training Dataset & Training Report
# Prepared By – Mandar Vilas Ukrulkar / 11-08-26
#  
# Name: dataset_v16v2_200726.clean.canonical.jsonl

Model target: Qwen3.5-4B financial-table-structure analyst (LoRA fine-tune, adapter `Qwen3.5-4B_v16v2_clean_dataset_210826_v2`)

Dataset file: dataset_v16v2_200726.clean.canonical.jsonl (6,109 records, ~100 MB)

Source corpus: C:\Users\mandar.ukrulkar\Downloads\data_raw_in_out_jsons_200726\output (the same job-output snapshot used to build the pre-canonicalization dataset)

Training code: VM_training_scripts/main_v2.py

Report generated: 2026-08-11

## 1. Executive summary

| Metric | Value |
|---|---|
| Source documents processed | 249 unique financial-statement documents (+3 re-run twice, +1 non-classification QA job) |
| Total tables detected by the parser | 19,821 |
| Tables classified as *financial* (candidates for labeling) | 6,260 |
| Tables skipped as non-financial (TOC, prose, boilerplate) | 13,532 (68%) |
| Raw compiled dataset (dataset_v16v2_200726.jsonl) | 6,289 records |
| De-duplicated clean dataset (*.clean.jsonl) | 6,109 records (180 removed, 2.9%) |
| **Canonicalized dataset (*.clean.canonical.jsonl)** | **6,109 records — 0 removed, value-level relabeling only** |
| Cells/fields remapped by canonicalization | table_type 420, concept_id 820, concept_meaning 2,742, unit 1,175, scale_multiplier 64,730 |
| Distinct `table_type` label strings | 2,087 (down from 2,232 pre-canonicalization, −6.5%) |
| Median table shape | 11 rows × 4 columns (unchanged by canonicalization) |
| Median prompt / completion length | 2,225 / 9,040 characters |
| Fine-tuning method | LoRA (r=32, alpha=64) on Qwen/Qwen3.5-4B, 1× NVIDIA L4 GPU, bf16 |
| Training run | 3 epochs / 363 steps, effective batch 48, ~11.4 hours |
| **Held-out evaluation** | **5% split (306 samples), evaluated every 40 steps — best checkpoint kept by eval_loss** |
| Final training loss (epoch 3 mean) | ~0.124 |
| Final eval loss / eval token accuracy | 0.137 / 96.55% |

The training set is built from the same real financial-statement tables (10-Ks, 20-Fs, annual reports, insurance/bank filings) as the `*.clean.jsonl` dataset, but with an added canonicalization pass that normalizes `table_type`, `concept_id`, `unit`, `scale_multiplier`, and `concept_meaning` values before fine-tuning. Unlike the sibling adapter trained on `*.clean.jsonl` (main.py, no eval split), this run holds out 5% of the data for genuine generalization tracking.

## 2. How this report was produced

Three data sources were combined:

- **Source job outputs** — C:\Users\mandar.ukrulkar\Downloads\data_raw_in_out_jsons_200726\output (253 job folders), the exact snapshot `dataprep_140726.py` compiled from. Unchanged from the `*.clean.jsonl` report — reproduced in §3 for a self-contained document.
- **The compiled/cleaned training files** — dataset_v16v2_200726.jsonl (raw) and dataset_v16v2_200726.clean.jsonl (de-duplicated), used here only as the "before" side of the canonicalization diff.
- **The canonical training file and its transform log** — dataprep/dataset_v16v2_200726.clean.canonical.jsonl (via `dataprep/canonicalize_dataset.py`) and dataprep/../report_assets/canonicalization_report.json, analyzed directly for §4 and the EDA in §5.

Provenance check: canonicalization changes field *values* inside existing records — it does not add, remove, or re-key records. `dataset_v16v2_200726.clean.canonical.jsonl` and `dataset_v16v2_200726.clean.jsonl` both contain exactly 6,109 records, confirmed by direct line count of both files. §3's job/document ledger therefore applies unchanged to this dataset.

## 3. Source corpus

### 3.1 Job-level overview

| | |
|---|---|
| Total processing jobs | 253 |
| — classification jobs (ai_provider: azure) | 251 |
| — non-classification QA/extraction jobs (no labeling) | 2 |
| Unique source documents | 249 (+3 documents each re-run through the pipeline twice) |
| Tables per job (table_count) — median / mean / max | 64 / 79.0 / 691 |
| Financial tables per job — median / mean / max | 11.5 / 25.0 / 222 |

Every classification job used the same pipeline configuration (ai_provider: azure, classified: true). The two non-classification jobs were extraction-only test runs against the same QA file and contributed zero labeled tables.

### 3.2 Documents by number of labeled financial tables

![top20 docs](report_assets_adapter1/top20_documents_by_tables_adapter1.png)

The corpus is dominated by long-form annual filings — 20-F filings and full annual reports (Eni, ENI SPA, Lavoro, Energy Company of Minas Gerais, Endesa, MC Group) each contribute 150–222 tables, an order of magnitude more than a typical single-country statutory filing (~10–30 tables).

### 3.3 Table-count distribution across the corpus

Heavily right-skewed: 62% of documents (156 of 250) contribute fewer than 20 financial tables; a long tail of 12 large multi-jurisdiction filings each contribute 100+.

## 4. From source tables to training set

```
Parser output (job dirs)         19,821 tables detected
    └─ financial-table filter →   6,260 tables classified as financial
         └─ dataprep_140726.py →  6,289 (prompt, completion) records  → dataset_v16v2_200726.jsonl
              └─ dataset_clean.py (near-duplicate table_type resolution)
                   └─ 6,109 records → dataset_v16v2_200726.clean.jsonl
                        └─ canonicalize_dataset.py (value-level normalization, no records dropped)
                             └─ 6,109 records → dataset_v16v2_200726.clean.canonical.jsonl   ← used for training
```

`canonicalize_dataset.py` does not touch record count — it relabels field *values* on the same 6,109 records that `dataset_clean.py` produced:

- **table_type**: 420 cells remapped, collapsing 2,232 → 2,087 distinct label strings.
- **concept_id**: 820 cells remapped, 67,473 → 67,134 distinct values.
- **concept_meaning**: 2,742 cells remapped — for a `concept_id` where one free-text meaning holds ≥60% share among its variants, all variants are canonicalized to that dominant meaning. 1,676 concept_ids were resolved this way; 17,211 remain ambiguous (no dominant meaning) and were left as-is — e.g. the concept_id `"Total"` alone has 365 distinct free-text meaning variants across 579 occurrences.
- **unit**: 1,175 cells remapped via a 3-entry alias map (`percentage`→`percent`, `EUR_thousand`/`thousand euro`→`thousand EUR`), reducing 210 → 188 distinct units. 32 additional mixed-scale unit variants (e.g. `million HUF`, `EUR_thousand`, `billion_cubic_meter`) were flagged but **not** auto-fixed and remain separate strings.
- **scale_multiplier**: 64,730 cells standardized down to 10 canonical multiplier values (`1000`, `1000000`, `1`, `0.01`, `1000000000`, `100`, and four rare fractional values).

## 5. EDA — canonical training dataset (6,109 records)

### 5.1 table_type label distribution

![top20 table types canonical](report_assets_adapter1/top20_table_types_canonical.png)

![canonicalization shrinkage](report_assets_adapter1/canonicalization_table_type_shrinkage.png)

- 2,087 distinct label strings across 6,109 records (down from 2,232 pre-canonicalization) — canonicalization closed 145 near-duplicate label variants, but the taxonomy is still effectively open-vocabulary/free-text rather than a fixed enum.
- `other` remains the single largest bucket, consistent with the pre-canonicalization distribution — tables the labeler couldn't map to a specific known statement type.
- Beyond the top ~20, the distribution still has a very long tail: most distinct labels occur only 1–3 times, so canonicalization narrows but does not solve the low-example-count problem for most specific table types.

### 5.2 Table shape

![table shape](report_assets_adapter1/table_shape_hist_adapter1.png)

| | min | p25 | median | mean | p75 | p90 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| Rows/table | 1 | 6 | 11 | 14.4 | 20 | 30 | 50 | 104 |
| Columns/table | 1 | 3 | 4 | 4.9 | 6 | 8 | 16 | 37 |
| Cells/table | 0 | 12 | 24 | 35.0 | 47 | 82 | 156 | 285 |

Identical to the pre-canonicalization dataset — canonicalization only rewrites `table_type`/`concept_id`/`unit`/`scale_multiplier`/`concept_meaning` values, not table dimensions.

### 5.3 Row and column semantic composition

![row type](report_assets_adapter1/row_type_distribution_adapter1.png)

![column type](report_assets_adapter1/column_type_distribution_adapter1.png)

Across all 6,109 tables, labeled rows total 88,154 and columns 30,083:

| Row type | Count | Share |
|---|---|---|
| data | 54,005 | 61.3% |
| row_header | 23,282 | 26.4% |
| total | 5,833 | 6.6% |
| subtotal | 4,153 | 4.7% |
| grand_total | 876 | 1.0% |
| change | 5 | <0.1% |

| Column type | Count | Share |
|---|---|---|
| period_data | 18,511 | 61.5% |
| label | 8,991 | 29.9% |
| total | 1,119 | 3.7% |
| note_ref | 820 | 2.7% |
| change | 441 | 1.5% |
| grand_total | 166 | 0.6% |
| subtotal | 35 | 0.1% |

Row/column type counts are byte-for-byte identical to the pre-canonicalization dataset, confirming `canonicalize_dataset.py` left structural row/column typing untouched.

### 5.4 Prompt / completion length

![length hist](report_assets_adapter1/prompt_completion_len_hist_adapter1.png)

| | min | p25 | median | mean | p75 | p90 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| Prompt (chars) | 443 | 1,437 | 2,225 | 2,859 | 3,720 | 5,472 | 10,055 | 17,648 |
| Completion (chars) | 427 | 4,807 | 9,040 | 12,721 | 17,255 | 28,522 | 52,758 | 86,231 |

Prompts are byte-identical to the pre-canonicalization file (prompts don't carry label values). Completions shift by a handful of characters at the median (9,035 → 9,040) — the net effect of swapping some label strings for slightly longer/shorter canonical equivalents (e.g. `EUR_thousand` → `thousand EUR`).

### 5.5 Label richness / confidence

- Cell `concept_id` fill rate: 93.0% — unchanged from the pre-canonicalization dataset; canonicalization relabels existing concept IDs, it doesn't add or remove them.
- Row `note_ref_value` fill rate: 7.4% — unchanged.
- `table_type` confidence — median 0.95, mean 0.93, min 0.20 — unchanged; canonicalization normalizes the label *value*, not the labeler's original confidence score.

## 6. Training run

Adapter: `Qwen3.5-4B_v16v2_clean_dataset_210826_v2` — a LoRA adapter fine-tuned on top of Qwen/Qwen3.5-4B using the canonical 6,109-record dataset from §5, via `VM_training_scripts/main_v2.py` (the sibling adapter trained on the pre-canonicalization `*.clean.jsonl` file used `main.py`).

### 6.1 Setup

| | |
|---|---|
| Base model | Qwen/Qwen3.5-4B |
| Method | LoRA (PEFT 0.20.0) |
| LoRA rank / alpha / dropout | r=32, alpha=64, dropout=0.05 |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, out_proj, in_proj_qkv, in_proj_a, in_proj_b, in_proj_z |
| Hardware | 1× NVIDIA L4 GPU |
| Precision | bf16, gradient checkpointing on |
| Software | torch 2.8.0+cu128, transformers 5.14.1 |
| Dataset | dataset_v16v2_200726.clean.canonical.jsonl — 5,803 train / **306 held-out eval (5%)**, max sequence length 1,024 tokens |
| Epochs | 3 |
| Per-device batch size / grad. accumulation | 2 / 24 → effective batch size 48 |
| Optimizer | AdamW (β1=0.9, β2=0.999, ε=1e-8), weight decay 0.01, max grad norm 1.0 |
| LR schedule | 5e-5 peak, cosine decay, 5% warmup |
| Eval strategy | every 40 steps on the held-out split; best checkpoint restored by `eval_loss` (`load_best_model_at_end=true`) |
| Seed | 42 |

The LoRA target list is identical to the sibling `main.py` run — same rank/alpha and the same gated-projection module names (`in_proj_qkv`, `in_proj_a/b/z`) alongside the standard q/k/v/o projections, consistent with Qwen3.5-4B's hybrid/gated attention block. The two runs differ in dataset (canonical vs. raw-clean), precision (bf16 vs. fp16), evaluation (held-out 5% split vs. none), and warmup fraction (5% vs. 3%) — not in LoRA capacity.

### 6.2 Loss curve

![loss curve](report_assets_adapter1/training_loss_curve_adapter1.png)

| Epoch | Mean train loss (logged steps) | Min | Max |
|---|---|---|---|
| 1 | 0.248 (skewed by the first few warmup steps) | 0.150 | 0.827 |
| 2 | 0.139 | 0.130 | 0.149 |
| 3 | 0.124 | 0.116 | 0.134 |

![eval loss and accuracy](report_assets_adapter1/eval_loss_accuracy_adapter1.png)

| Step | Epoch | Eval loss | Eval token accuracy |
|---|---|---|---|
| 40 | 0.33 | 0.184 | 95.63% |
| 80 | 0.66 | 0.162 | 96.04% |
| 120 | 0.99 | 0.152 | 96.23% |
| 160 | 1.32 | 0.146 | 96.35% |
| 200 | 1.65 | 0.142 | 96.44% |
| 240 | 1.98 | 0.139 | 96.48% |
| 280 | 2.31 | 0.138 | 96.51% |
| 320 | 2.65 | 0.137 | 96.53% |
| 360 | 2.98 | 0.137 | 96.54% |
| 363 (final) | 3.00 | 0.137 | 96.55% |

Because this run has a genuine held-out eval split, both curves can be checked against each other directly. Train and eval loss fall together throughout the run with no divergence — the eval curve is never meaningfully above the train curve, and the best checkpoint by `eval_loss` is the final step (363), so there is no overfitting signal across the full 3 epochs. Loss drops fastest in epoch 1 as the model learns the rigid JSON output template, then flattens into a floor around 0.137 (eval) / 0.124 (train) by epoch 2, with only marginal further gains in epoch 3 — expected diminishing-returns behavior under cosine LR decay, not a stall. Eval token accuracy rises in lockstep from 95.6% to 96.55%. The warmup spike here (loss 0.83 at step 1) is far milder than the sibling `main.py` run's (~4.3, with `grad_norm: NaN` for the first 3 logged steps) — this run trains in bf16 rather than fp16 and never hits that early instability, so the trainer-reported aggregate `train_loss` (0.124) closely tracks the epoch-3 mean rather than being skewed by a warmup spike.

### 6.3 Learning-rate schedule

![lr schedule](report_assets_adapter1/training_lr_schedule_adapter1.png)

Warmup over the first 5% of steps to a 5e-5 peak, followed by cosine decay to ~0 over the remaining steps — visible directly in the loss curve's soft convergence across epochs 2–3.

### 6.4 Compute

| | |
|---|---|
| Total steps | 363 (121/epoch × 3 epochs) |
| Total training time | 41,177 s ≈ 11.4 hours |
| Throughput | 0.423 samples/s, 0.009 steps/s |
| Total FLOs | 3.88 × 10¹⁷ |
| Checkpointing | every 40 steps, best 3 kept; best-by-eval_loss checkpoint (step 363) restored at end |

A single L4 GPU, bf16 + gradient checkpointing, pushing an effective batch of 48 through 1,024-token sequences for ~11.4 hours to complete 3 epochs over the 6,109-record set plus a 306-sample eval pass every 40 steps — a modest, single-GPU LoRA fine-tune, faster wall-clock than the sibling `main.py` run despite the added eval overhead (14.7 hours), consistent with bf16 vs. fp16 throughput differences on this hardware.

## 7. Notes and caveats

- Canonicalization is a value-level relabeling pass over the exact same 6,109 clean records — no records were added or removed, so §3's corpus/job ledger from the `*.clean.jsonl` report carries over to this dataset unchanged.
- `table_type` is still a long-tail, open-vocabulary taxonomy after canonicalization: 2,087 distinct strings across 6,109 records. Canonicalization closed 145 near-duplicate variants (2,232→2,087) via 420 remapped cells, but did not collapse the taxonomy to a fixed enum.
- `concept_meaning` canonicalization only resolves `concept_id`s with a dominant (≥60%) meaning share; 17,211 concept_ids remain ambiguous and were left as free text — a candidate area for further normalization if concept-level accuracy matters more than structural accuracy.
- Unit canonicalization applied only a 3-entry alias map; 32 additional mixed-scale unit variants (several hundred cells each, e.g. `million HUF`, `EUR_thousand`) were flagged but not auto-fixed and remain as separate unit strings in the training data.
- This run used a genuine 5% held-out eval split (306 samples) with `load_best_model_at_end=true` — reported metrics reflect real generalization, not just training-set fit. Contrast with the sibling `*.clean.jsonl` / `main.py` run, which had `eval_strategy: no` and reports training loss only.
- The train/eval loss floor here (~0.124 / ~0.137) is notably higher than the sibling run's near-zero training loss (~0.025–0.03). Given the sibling run had no eval split, its lower number most likely reflects the raw (non-deduplicated-for-label-consistency) file being easier to fit/memorize, with no held-out check to catch that — it is not evidence that run generalizes better.
- Full raw statistics backing this report are in `report_assets_adapter1/stats_adapter1.json`; regenerate with `report_assets_adapter1/analyze_adapter1.py` (dataset stats), `report_assets_adapter1/chart_gen_adapter1.py` (dataset charts), and `report_assets_adapter1/train_chart_gen_adapter1.py` (training-run charts).
